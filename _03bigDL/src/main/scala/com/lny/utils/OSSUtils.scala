package com.lny.utils

import java.util

import com.aliyun.oss.model._
import com.aliyun.oss.{OSS, OSSClientBuilder}

object OSSUtils {
  /**
    * 获取 OssClient
    * @return
    */
  def getOss(): OSS = {
    val endpoint: String = "https://oss-cn-shanghai-internal.aliyuncs.com"
    val accessKeyId = "LTAIQENhPMB5rMgl"
    val accessKeySecret = "AgCXsxreFgjXc12KhJtUtrRJNcV40T"
    val bucketName = "dtbird-platform"
    val ossClient: OSS = new OSSClientBuilder().build(endpoint, accessKeyId, accessKeySecret)

    ossClient

  }


  /**
    * 判断文件夹是否存在
    * @param path
    * @return
    */
  def isOssDirExist(path: String): Boolean = {
    val endpoint: String = "http://oss-cn-shanghai-internal.aliyuncs.com"
    val accessKeyId = "LTAIQENhPMB5rMgl"
    val accessKeySecret = "AgCXsxreFgjXc12KhJtUtrRJNcV40T"
    val bucketName = "dtbird-platform"
    val ossClient: OSS = new OSSClientBuilder().build(endpoint, accessKeyId, accessKeySecret)

    println("========================================")
    val request = new GenericRequest(bucketName, path)
    val found: Boolean = ossClient.doesObjectExist(request)
    // 关闭OSSClient。
    println(found)
    ossClient.shutdown()
    found
  }

  /**
    * get all dirs right down the specific path
    * 获取某个path下的所有文件夹
    *
    */
  def listOssDirs(path: String): util.List[String] = {
    val bucketName = "dtbird-platform"
    val ossClient: OSS = getOss()
    // 构造ListObjectsRequest请求。
    val listObjectsRequest = new ListObjectsRequest(bucketName)

    // 设置正斜线（/）为文件夹的分隔符。
    listObjectsRequest.setDelimiter("/")
    // 列出fun目录下的所有文件和文件夹。
    listObjectsRequest.setPrefix(path)

    val listfile: ObjectListing = ossClient.listObjects(listObjectsRequest)
    ossClient.shutdown()
    listfile.getCommonPrefixes

  }
  /**
    * get all files&dirs right down the specific path
    * 获取某个path下的所有文件和文件夹
    *
    */
  def listOssFiles(ossClient:OSS,path: String): util.List[OSSObjectSummary] = {
    val bucketName = "dtbird-platform"
    //得到目录下的文件和文件夹
    val listing: ObjectListing = ossClient.listObjects(bucketName,path)
    val summaries: util.List[OSSObjectSummary] = listing.getObjectSummaries
    summaries

  }

  /**
    * 删除oss文件
    * @param path
    */
  def deleteOssFile(path:String)={
    val bucketName = "dtbird-platform"
    //获取client
    val ossClient: OSS = getOss()
    //删除文件
    val request = new GenericRequest(bucketName,path)
    ossClient.deleteObject(request)
    //close
    ossClient.shutdown()
  }


  /**
    * 删除oss文件夹
    * @param path
    */
  def deleteOssDir(path:String)={
    import scala.collection.JavaConverters._
    val bucketName = "dtbird-platform"
    //获取client
    val ossClient: OSS = getOss()
    val summaries: util.List[OSSObjectSummary] = listOssFiles(ossClient,path)
    val strs = new util.ArrayList[String]()
    for (elem <- summaries.asScala) {
      strs.add(elem.getKey)
    }
    val deleteRequest: DeleteObjectsRequest = new DeleteObjectsRequest(bucketName).withKeys(strs)
    ossClient.deleteObjects(deleteRequest)
    ossClient.deleteObject(bucketName,path)
    //close
    ossClient.shutdown()
  }

}
